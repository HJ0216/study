//import java.util.Arrays;

public class SelectionSort { // 선택 정렬
	// Arrays.sort(ar); ascending_sort
	
	public static void main(String[] args) {
//		int[] ar = {32, 40, 25, 78, 56};
		int[] ar = {78, 56, 40, 32, 25};

		for(int i = 0; i<ar.length; i++) {
			System.out.print(String.format("%4d ", ar[i])); // 4자리 10진수값 지정
			// %x: 16진수, %d: 10진수, %s: String
			// System.out.printf("%4d", ar[i]);
		}
		
		System.out.println();

		for(int j=0; j<ar.length-1; j++) { // selection sort: 배열의 최종값은 비교 주체 X
			for(int k=j+1; k<ar.length; k++) { // selection sort: 배열의 최초값은 비교 객체X
				if(ar[j]>ar[k]) { // >: ascending, <: descending
					int tmp = ar[k];
					ar[k] = ar[j];
					ar[j] = tmp;
				}
			} // for k
		} // for j

		/*
		 j = 0 -> k = 1 2 3 4
		 j = 1 -> k = 2 3 4
		 j = 2 -> k = 3 4
		 j = 3 -> k = 4
		 */

		for(int i = 0; i<ar.length; i++) {
			System.out.print(String.format("%4d ", ar[i])); // 4자리 10진수값 지정
			// %b: boolean, %o: 8진수 Integer, %d: 10진수 Integer, %x: 16진수 Integer, %f: float, %s: String
			// System.out.printf("%4d", ar[i]);
		}
		
		
	}
	
}
