public class DoWhile {

	public static void main(String[] args) {
		int a = 0;
		
		do {
			a++;
			System.out.print(a + "  ");
		} while(a<10); // ; 유의
		
		System.out.println();
		System.out.println();
		
		char ch = 'A';
		int count = 0;
		
		do {
			System.out.print((ch++)+ "  ");
			
			count++;
			if(count%7==0) { // n의 배수: x%n==0
				System.out.println();
				} // if
		} while(ch<='Z'); // do-while
		
	}
	
}
